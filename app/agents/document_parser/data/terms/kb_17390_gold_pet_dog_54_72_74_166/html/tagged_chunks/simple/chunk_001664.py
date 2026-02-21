from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>154 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><p id='7' "
 "data-category='list' style='font-size:16px'>항목 이상에서 독립적 수행이 불가능하여 타인의 도움이 "
 '필요하고<br>GAF 70점 이하인 상태를 말한다.<br>아) 지속적인 정신건강의학과의 치료란 3개월 이상 약물치료가 '
 '중단되지<br>않았음을 의미한다.<br>자) 심리학적 평가보고서는 정신건강의학과 의료기관에서 실시되어져야<br>하며, 자격을 갖춘 '
 '임상심리전문가가 시행하고'),
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
 'indexing': {'chunk_id': 'chunk_001664',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
