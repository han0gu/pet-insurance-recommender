from langchain_core.documents import Document

chunk = Document(
    page_content=(". 위험이 뚜렷이 변경되거나 변경되었음을 알았을 때</p><br><p id='93' data-category='list' "
 "style='font-size:14px'>② 회사는 제1항에 따라 위험이 감소된 경우에는 그 차액보험료를 돌려드리며, 위험이 "
 '증<br>가된 경우에는 통지를 받은 날부터 1개월 이내에 보험료의 증액을 청구하거나 계약을<br>해지할 수 있습니다.<br>③ 계약자 '
 '또는 피보험자는 주소 또는 연락처가 변경된 경우에는 지체없이 이를 회사에 알<br>려야 합니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000261',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
