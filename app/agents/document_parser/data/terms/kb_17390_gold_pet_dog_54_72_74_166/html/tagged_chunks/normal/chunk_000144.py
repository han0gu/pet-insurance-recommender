from langchain_core.documents import Document

chunk = Document(
    page_content=(". 부담보가 지정된 질병 또는 증상이 악화되지 않고 유지된 경우</p><br><p id='172' "
 'data-category=\'paragraph\' style=\'font-size:14px\'>\uf000 제5항의 "청약일로부터 5년이 '
 '지나는 동안"이라 함은 제28조(보험료의 납입이 연</p><br><p id=\'173\' '
 "data-category='paragraph' style='font-size:14px'>체되는 경우 납입최고(독촉)와 계약의 해지)에서 "
 "정한 계약의 해지가 발생하지 않</p><br><h1 id='174'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000144',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
