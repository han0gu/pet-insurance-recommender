from langchain_core.documents import Document

chunk = Document(
    page_content=('- 발생한 경우에는 보장합니다)\n'
 '- 3. 선박에 탑승하는 것을 직무로 하는 사람이 직무상 선박에 탑승하고 있는 동안\n'
 '# <용어풀이># [흥행]영리를 목적으로 연극, 영화, 서커스 등을 요금을 받고 대중에게 보여주는 행위를 말합니다.제8조 (보험금 '
 '지급사유의 통지)계약자 또는 피보험자나 보험수익자는 제3조(보험금의 지급사유)에서 정한 보험금 지급\n'
 '사유의 발생을 안 때에는 지체없이 그 사실을 회사에 알려야 합니다.# 제 9조 (보험금 등의 청구)① 보험수익자는 다음의 서류를 제출하고 '
 '보험금 또는 보험료 납입면제를 청구하여야 합'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000165',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
