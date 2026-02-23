from langchain_core.documents import Document

chunk = Document(
    page_content=('는 것을 말합니다. 단 수술에서 아래에 정한 사항은 제외합니다- 1. 흡인 (주사기 등으로 빨아 들이는 것)\n'
 '- 2. 천자 (바늘 또는 관을 꽂아 체액, 조직을 뽑아내거나 약물을 주입하는 것) 등의 조치\n'
 '- 3. 미용성형 목적의 수술\n'
 '# 4. 검사 및 진단을 위한 수술 (생검, 복강경 검사)# 제 5조 (보험금의 청구)# ① 보험수익자는 다음의 서류를 제출하고 보험금을 '
 '청구하여야 합니다.- 1. 보험금 청구서(회사 양식)\n'
 '- 2. 등록견의 경우에는 동물등록증 또는 등록번호'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000570',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
