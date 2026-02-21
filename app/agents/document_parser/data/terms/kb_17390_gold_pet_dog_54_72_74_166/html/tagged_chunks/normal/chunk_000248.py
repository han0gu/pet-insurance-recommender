from langchain_core.documents import Document

chunk = Document(
    page_content=("id='57' data-category='paragraph' style='font-size:16px'>\uf000 제28조(보험료의 "
 "납입이 연체되는 경우 납입최고(독촉)와 계약의 해지)에 따라 계</p><br><p id='58' "
 "data-category='paragraph' style='font-size:16px'>약이 해지되었으나 해약환급금을 받지 않은 "
 '경우(보험계약대출 등에 따라 해약환<br>급금이 차감되었으나 받지 않은 경우 또는 해약환급금이 없는 경우를 '
 '포함합니다)<br>공<br>계약자는 해지된 날부터 3년 이내에 회사가 정한'),
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
 'indexing': {'chunk_id': 'chunk_000248',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
