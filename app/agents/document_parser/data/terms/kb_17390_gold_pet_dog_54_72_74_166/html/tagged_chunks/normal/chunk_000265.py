from langchain_core.documents import Document

chunk = Document(
    page_content=("제1항에 따른 해약환급금을 계약자<br>에게 지급합니다.</p><br><p id='71' data-category='paragraph' "
 "style='font-size:14px'>제31조의1(위법계약의 해지)</p><br><h1 id='72' "
 "style='font-size:14px'>\uf000 계약자는</h1><br><p id='73' "
 'data-category=\'paragraph\' style=\'font-size:14px\'>"금융소비자보호에 관한 법률" 제47조 및 '
 "관련규정이 정하는 바에 따라</p><br><p id='74'"),
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
 'indexing': {'chunk_id': 'chunk_000265',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
