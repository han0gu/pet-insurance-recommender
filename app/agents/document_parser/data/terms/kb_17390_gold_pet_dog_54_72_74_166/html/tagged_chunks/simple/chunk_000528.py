from langchain_core.documents import Document

chunk = Document(
    page_content=('받은 경우 이 특별약관의 보험가입금액을 연간 1회에<br>한하여 보험수익자에게 지급합니다.<br>\uf000 제1항에서 "연간"이란 '
 "계약일로부터 매1년 단위로 도래하는 계약해당일 전까지</p><br><p id='272' "
 "data-category='list'></p><br><p id='273' data-category='paragraph' "
 "style='font-size:14px'>특</p><p id='274' data-category='paragraph' "
 "style='font-size:16px'>기간을 의미합니다.</p><p id='275'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000528',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
