from langchain_core.documents import Document

chunk = Document(
    page_content=('5월 1일<br>- 자기부담금: 3만원<br>- 보상비율: 70%<br>- 1일 보상한도: 20만원<br>- 연간 총보상한도: '
 "30만원</p><br><table id='30' "
 "style='font-size:18px'><thead></thead><tbody><tr><td>입˙통원 "
 '진료일</td><td>2025.5.1</td><td>2025.8.1.</td><td>2025.11.1</td></tr><tr><td>피보험자가 '
 '부담한'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000022',
              'chunk_char_len': 239,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
