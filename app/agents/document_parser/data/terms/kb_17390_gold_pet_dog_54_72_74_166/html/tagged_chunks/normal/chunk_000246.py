from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제3호 및 제4호의 내용에 관한 사항을 계약자에게 안내할 것<br>\uf000 제1항에 따라 계약이 해지된 경우에는 '
 "제34조(해약환급금) 제1항에 따른 해약환급</p><br><p id='54' data-category='paragraph' "
 "style='font-size:16px'>금을 계약자에게 지급합니다.<br>용 어 풀 이 납입최고(독촉)</p><br><table "
 "id='55' style='font-size:16px'><thead></thead><tbody><tr><td>약정된 "
 '기일까지</td><td>보험료가'),
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
 'indexing': {'chunk_id': 'chunk_000246',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
