from langchain_core.documents import Document

chunk = Document(
    page_content=('풀 이 납입최고(독촉) 약정된 기일까지 보험료가 납입되지 않을 경우, 회사가 계약자에게 '
 "보험료의</td></tr></tbody></table><br><h1 id='64' style='font-size:16px'>납입을 "
 "재촉하는 것을 말합니다.</h1><p id='65' data-category='list' "
 "style='font-size:16px'>제17조(보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))<br>\uf000 "
 '제16조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 특별약관의 해지)에 따<br>라 특별약관이'),
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
 'indexing': {'chunk_id': 'chunk_000889',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
