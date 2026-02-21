from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 제2항에서 정한 조치에 다른 진료를 병행하여 실시한 경우에는 제2항에서 정한\n'
 '조치(마취 비용을 포함합니다.)에 대한 보험금은 지급하지 않습니다.# 제7조 (보험금 지급사유의 통지)계약자 또는 피보험자나 보험수익자는 '
 '제3조(보험금의 지급사유)에서 정한 보험금 지급\n'
 '사유의 발생을 안 때에는 지체없이 그 사실을 회사에 알려야 합니다.# 제8조 (보험금의 청구)① 보험수익자는 다음의 서류를 제출하고 '
 '보험금을 청구하여야 합니다.- 1. 보험금 청구서(회사 양식)\n'
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
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000478',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
