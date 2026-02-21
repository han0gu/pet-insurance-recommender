from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회 사가 변경 전 보험수익자에게 보험금을 지급한 경우 변경된 보험수익자에게는 별도로 보험금을 지급하지 '
 "않습니다.</td></tr></tbody></table><br><p id='10' data-category='list' "
 "style='font-size:14px'>\uf000 회사는 계약자가 제1회 보험료를 납입한 때부터 1년 이상 지난 유효한 "
 '계약으로서<br>그 보험종목의 변경을 요청할 때에는 회사의 사업방법서에서 정하는 방법에 따라<br>이를 변경하여 '
 '드립니다.<br>\uf000 회사는 계약자가 제1항 제5호에 따라 보험가입금액 또는'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000195',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
