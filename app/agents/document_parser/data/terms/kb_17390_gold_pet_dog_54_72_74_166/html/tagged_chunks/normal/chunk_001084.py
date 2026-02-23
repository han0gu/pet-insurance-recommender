from langchain_core.documents import Document

chunk = Document(
    page_content=(". 계약자, 피보험자, 이들의 가족 또는 사용인의</p><br><p id='64' data-category='paragraph' "
 "style='font-size:16px'>고의 또는 중대한 과실</p><br><p id='65' data-category='list' "
 "style='font-size:16px'>2. 전쟁, 혁명, 내란, 사변, 테러, 폭동, 소요, 노동쟁의, 기타 이들과 "
 '유사한<br>사태<br>3. 지진, 분화, 홍수, 해일 또는 이와 비슷한 천재지변<br>4'),
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
 'indexing': {'chunk_id': 'chunk_001084',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
