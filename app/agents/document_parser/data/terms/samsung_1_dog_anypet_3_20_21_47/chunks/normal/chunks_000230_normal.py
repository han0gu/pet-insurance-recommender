from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 파보바이러스 감염증 2. 디스템퍼바이러스 감염증 3. 코로나바이러스 감염증\n'
 '【펫샵】 동물보호법 시행규칙에 따라 동물을 분양하는 영업활동을 할 수 있는 영업자를 말합니다. 【분양】 펫샵에 유상의 재화를 제공하고 '
 '반려동물을 입양하는 행위를 말합니다.\n'
 '② 반려동물이 제1항의 사고로 치료를 받던 중에 보험개시일로부터 30일이 지난 경우에도 보험개시일 부터 120일 이내의 치료비는 보상하여 '
 '드립니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 47},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000230',
              'chunk_char_len': 223,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
