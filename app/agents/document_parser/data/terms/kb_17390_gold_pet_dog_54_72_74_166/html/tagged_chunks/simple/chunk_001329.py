from langchain_core.documents import Document

chunk = Document(
    page_content=("id='192' style='font-size:14px'>행한 사고처리 확인원등으로 결정합니다.</h1><p id='193' "
 "data-category='paragraph' style='font-size:14px'>제3조(해지된 특약의 "
 '부활(효력회복))<br>회사는 이 특별약관의 부활(효력회복)청약을 받은 경우에는 보험계약의 부활(효력회<br>복)을 승낙한 경우에 한하여 '
 '보통약관 제1절 일반조항 제29조(보험료의 납입을 연체<br>하여 해지된 계약의 부활(효력회복)) 및 제30조(강제집행 등으로 인하여 '
 '해지된 계약<br>의'),
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
 'indexing': {'chunk_id': 'chunk_001329',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
