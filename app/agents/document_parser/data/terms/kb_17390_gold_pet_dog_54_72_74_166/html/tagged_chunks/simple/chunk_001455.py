from langchain_core.documents import Document

chunk = Document(
    page_content=(". 제1항에서 지정한 특정질병으로 인하여 사망하여 보험금의 지급사유가 발생<br>한 경우</p><br><p id='124' "
 "data-category='list' style='font-size:16px'>\uf000 해당 반려동물에게 보험사고가 발생했을 경우, "
 '그 사고가 특정질병을 직접적인 원<br>인으로 발생한 사고인가 아닌가는 수의사의 진단서와 의견을 주된 판단자료로 '
 "결<br>정합니다.<br>\uf000 제1항의 특정질병은 1개에 한하여 부가할 수 있습니다.</p><br><p id='125' "
 "data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_001455',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
