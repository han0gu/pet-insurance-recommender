from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사가 변경<br>전 보험수익자에게 보험금을 지급한 경우 변경된 보험수익자에게는 별도로 보험금을<br>지급하지 '
 "않습니다.</p><br><p id='36' data-category='list' style='font-size:14px'>③ 회사는 "
 '계약자가 제1회 보험료를 납입한 때부터 1년 이상 지난 유효한 계약으로서 그<br>보험종목의 변경을 요청할 때에는 회사의 사업방법서에서 '
 '정하는 방법에 따라 이를 변<br>경하여 드립니다.<br>④ 회사는 계약자가 제1항 제5호에 따라 보험가입금액을 감액하고자 할 때에는 그'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000132',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
