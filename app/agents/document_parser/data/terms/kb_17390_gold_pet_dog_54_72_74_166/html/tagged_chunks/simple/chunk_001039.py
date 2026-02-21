from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약자, 피보험자, 이들의 가족 또는 사용인의 고의 또는 중대한 과실<br>2. 지진, 분화, 해일, 홍수 또는 이와 유사한 '
 '자연재해로 생긴 손해<br>3. 전쟁, 외국의 무력행사, 혁명, 내란, 사변, 폭동, 소요, 기타 이들과 유사한<br>사태<br>4. '
 '핵연료 물질 또는 핵연료 물질에 의하여 오염된 물질의 방사성, 폭발성 또는<br>그 밖의 유해한 특성에 의한 사고<br>5. 위 제4호 '
 '이외의 방사선을 쬐는 것 또는 방사능 오염<br>6. 최초계약의 보험계약일 이전에 이미 감염 또는 발병한 질병 및 상해<br>7'),
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
 'indexing': {'chunk_id': 'chunk_001039',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
