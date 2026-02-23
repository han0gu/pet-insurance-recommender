from langchain_core.documents import Document

chunk = Document(
    page_content=('년 동일한 금액으로 나누어 지급받는 경우| 지급일 | 지급액 |\n'
 '| --- | --- |\n'
 '| 2024년 4월 1일 | 2천만원 |\n'
 '| 2025년 4월 1일 | 2천만원 × (1 + 평균공시이율) |\n'
 '| 2026년 4월 1일 | 2천만원 × (1 + 평균공시이율)2 |\n'
 '# 제10조 (주소변경통지)① 계약자(보험수익자가 계약자와 다른 경우 보험수익자를 포함합니다)는 주소 또는 연락\n'
 '처가 변경된 경우에는 지체없이 그 변경내용을 회사에 알려야 합니다.에 대한 기록이 남는 방법으로 회사가 알린 사항은 일반적으로 도달에 '
 '필요한 기간이'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000029',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
