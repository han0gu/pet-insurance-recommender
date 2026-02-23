from langchain_core.documents import Document

chunk = Document(
    page_content=('정산금액(이하 "정산금액"이라 합니다)을 환급하여 드립니다. 한편 위험이 증가\n'
 '된 경우에는 보험료의 증액 및 정산금액의 추가납입을 요구할 수 있으며, 계약자\n'
 '는 일시납 또는 잔여 보험료 납입기간과 5년 중 큰 기간(단, 잔여 보험기간을 초\n'
 '과할 수 없음) 동안의 분납 중 선택하여 정산금액을 납입하여야 합니다. 다만, 보\n'
 '험료 갱신형 계약 등 일부 보험계약의 경우 분납이 제한될 수 있습니다.\n'
 '\uf000 제1항의 통지에 따라 위험의 증가로 보험료를 더 내야 할 경우 회사가 청구한 추가'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000066',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
