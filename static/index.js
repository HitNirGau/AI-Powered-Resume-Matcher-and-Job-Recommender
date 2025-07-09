const jobPostsData = [
  {
    title: "Google Summer of Code",
    description: "Work on open-source projects globally.",
    area: "Pune",
    category: "Frontend Developer",
    moreInfoLink: "https://summerofcode.withgoogle.com/get-started",
    applyLink: "https://summerofcode.withgoogle.com/get-started"
  },
  {
    title: "Microsoft Internship",
    description: "Tech-focused internship program.",
    area: "Gurgaon",
    category: "Backend Developer",
    moreInfoLink: "https://careers.microsoft.com/students/us/en",
    applyLink: "https://careers.microsoft.com/students/us/en"
  },
  {
    title: "Amazon Internship",
    description: "Internship at Amazon for developers.",
    area: "Hyderabad",
    category: "Full Stack Developer",
    moreInfoLink: "https://www.amazon.jobs/en/job_categories/internships",
    applyLink: "https://www.amazon.jobs/en/job_categories/internships"
  },
  {
    title: "Tesla AI Research Internship",
    description: "Work on AI research at Tesla.",
    area: "Bangalore",
    category: "Flutter Developer",
    moreInfoLink: "https://www.tesla.com/careers/search",
    applyLink: "https://www.tesla.com/careers/search"
  }
  // Add more job posts here as needed...
];

document.getElementById("searchForm").addEventListener("submit", function(event) {
  event.preventDefault(); // Prevent page reload on form submission

  const searchTerm = document.getElementById("searchInput").value.toLowerCase();
  const selectedArea = document.getElementById("areaSelect").value;
  const selectedCategory = document.getElementById("categorySelect").value;

  // Filter job posts based on the search criteria
  const filteredPosts = jobPostsData.filter(post => {
    const matchesSearchTerm = post.title.toLowerCase().includes(searchTerm);
    const matchesArea = selectedArea === "Select Area" || post.area === selectedArea;
    const matchesCategory = selectedCategory === "Select Category" || post.category === selectedCategory;

    return matchesSearchTerm && matchesArea && matchesCategory;
  });

  displayFilteredPosts(filteredPosts);
});

function displayFilteredPosts(posts) {
  const jobPostsContainer = document.getElementById("jobPosts");
  jobPostsContainer.innerHTML = ""; // Clear any previous results

  if (posts.length === 0) {
    jobPostsContainer.innerHTML = "<p>No jobs found based on your search criteria.</p>";
  } else {
    posts.forEach(post => {
      const postElement = document.createElement("div");
      postElement.classList.add("col-md-4", "mb-4");

      postElement.innerHTML = `
        <div class="card p-3">
          <h3>${post.title}</h3>
          <p>${post.description}</p>
          <a href="${post.moreInfoLink}" class="btn btn-info mb-3" target="_blank">More Info</a>
          <a href="${post.applyLink}" class="btn btn-primary" target="_blank">Apply Now</a>
        </div>
      `;
      jobPostsContainer.appendChild(postElement);
    });
  }
}