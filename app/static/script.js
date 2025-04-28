const uploadArea = document.getElementById('uploadArea');
const uploadInput = document.getElementById('uploadInput');
const generateBtn = document.getElementById('generateBtn');
const loading = document.getElementById('loading');
const captionCards = document.getElementById('captionCards');

let uploadedImage = null;

// Xử lý sự kiện click vào khu vực upload
uploadArea.addEventListener('click', () => {
    uploadInput.click();
});

// Ngăn chặn hành vi mặc định khi kéo thả
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

// Xử lý sự kiện khi thả file vào vùng upload
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');

    if (e.dataTransfer.files.length > 0) {
        uploadedImage = e.dataTransfer.files[0];
        displayImage(uploadedImage);
        generateBtn.disabled = false;
    }
});

// Xử lý sự kiện khi file được chọn qua input
uploadInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        uploadedImage = e.target.files[0];
        displayImage(uploadedImage);
        generateBtn.disabled = false;
    }
});

// Hiển thị hình ảnh đã tải lên
function displayImage(file) {
    const reader = new FileReader();

    reader.onload = (e) => {
        uploadArea.innerHTML = '';
        uploadArea.classList.add('has-image');

        const img = document.createElement('img');
        img.src = e.target.result;
        uploadArea.appendChild(img);
    };

    reader.readAsDataURL(file);
}

// Gửi ảnh đến Flask để sinh caption
generateBtn.addEventListener('click', () => {
    if (!uploadedImage) return;

    // Hiển thị loading
    loading.style.display = 'block';
    generateBtn.disabled = true;

    // Tạo formData để gửi file
    let formData = new FormData();
    formData.append('image', uploadedImage);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            loading.style.display = 'none';
            generateBtn.disabled = false;
            displayCaptions(data.captions); // Cập nhật để xử lý danh sách captions
        })
        .catch(error => {
            console.error("Lỗi khi gọi API:", error);
            loading.style.display = 'none';
            generateBtn.disabled = false;
            alert("Có lỗi xảy ra. Vui lòng thử lại.");
        });
});

// Hiển thị danh sách 5 captions từ Flask
function displayCaptions(captions) {
    captionCards.innerHTML = ''; // Xóa các card cũ

    captions.forEach((caption, index) => {
        const card = document.createElement('div');
        card.className = 'caption-card';

        const captionText = document.createElement('p');
        captionText.className = 'caption-text';
        captionText.textContent = `${index + 1}. ${caption}`; // Thêm số thứ tự

        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-btn';
        copyBtn.textContent = 'Copy';
        copyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(caption).then(() => { // Chỉ copy nội dung caption, không copy số thứ tự
                copyBtn.textContent = 'Copied';
                copyBtn.classList.add('copied');
                setTimeout(() => {
                    copyBtn.textContent = 'Copy';
                    copyBtn.classList.remove('copied');
                }, 2000);
            });
        });

        card.appendChild(captionText);
        card.appendChild(copyBtn);
        captionCards.appendChild(card);
    });
}