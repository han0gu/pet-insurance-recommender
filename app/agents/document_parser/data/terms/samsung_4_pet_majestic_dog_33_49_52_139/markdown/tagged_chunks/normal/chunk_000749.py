from langchain_core.documents import Document

chunk = Document(
    page_content=('- 정을 적용합니다.\n'
 '# 제2조 (보험료의 영수)자동납입일자 또는 급여이체일자는 이 청약서에 기재된 보험료납입 해당일에도 불구하고\n'
 '회사와 계약자가 별도로 약정한 일자로 합니다.# 제 3조 (계약 후 알릴 의무)계약자는 지정계좌의 번호가 변경 또는 거래정지된 경우에는 '
 '그 사실을 즉시 회사에 알'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000749',
              'chunk_char_len': 163,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
