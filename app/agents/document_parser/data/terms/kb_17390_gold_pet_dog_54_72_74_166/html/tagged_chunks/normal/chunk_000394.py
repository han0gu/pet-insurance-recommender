from langchain_core.documents import Document

chunk = Document(
    page_content=('35%에서 질병으로 인한 장해지급률 15%를 차감한 지급률 20%(=35%-15%)에 해당하는 후유장해보험금을 '
 "지급</td></tr></tbody></table><br><p id='54' data-category='paragraph' "
 "style='font-size:14px'>\uf000 회사가 지급하여야 할 하나의 상해로 인한 후유장해보험금은 보험가입금액을 "
 "한</p><br><p id='55' data-category='paragraph' style='font-size:14px'>도로 "
 "합니다.</p><br><h1 id='56'"),
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
 'indexing': {'chunk_id': 'chunk_000394',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
