from langchain_core.documents import Document

chunk = Document(
    page_content=("/></figure><br><p id='192' data-category='paragraph' "
 "style='font-size:14px'>\uf000 이 계약이 의무보험이 아니고 다른 의무보험이 있는 경우에는 다른 "
 '의무보험에서<br>보상되는 금액(피보험자가 가입을 하지 않은 경우에는 보상될 것으로 추정되는<br>금액)을 차감한 금액을 손해액으로 '
 '간주하여 제1항에 의한 보상할 금액을 결정합<br>니다.<br>\uf000 피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 '
 '회사의 제1항에<br>의한 지급보험금 결정에는 영향을 미치지'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001177',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
