from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 회사는 제2항에 따라 계약내용을 변경할 때 위험이 감소된 경우에는 보험료를 감액하 고, 이후 기간 보장을 위한 재원인 계약자적립액 '
 '등의 차이로 인하여 발생한 정산금액 (이하 「정산금액」이라 합니다)을 환급하여 드립니다. 한편 위험이 증가된 경우에는 보험료의 증액 및 '
 '정산금액의 추가납입을 요구할 수 있으며, 계약자는 이를 납입하여 야 합니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 71},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000388',
              'chunk_char_len': 192,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
