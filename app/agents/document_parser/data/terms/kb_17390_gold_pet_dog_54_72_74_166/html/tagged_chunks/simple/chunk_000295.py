from langchain_core.documents import Document

chunk = Document(
    page_content=('. 또한, 지정대리청구인은 제40조(지정대리청구인의 변경지정)에 의한 변경지정<br>별<br>또는 보험금 청구 시에도 다음 각 호의 어느 '
 "하나에 해당하여야 합니다. 표</p><br><p id='126' data-category='list' "
 "style='font-size:16px'>1. 피보험자의 가족관계등록부상 또는 주민등록상의 배우자<br>2"),
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
 'indexing': {'chunk_id': 'chunk_000295',
              'chunk_char_len': 193,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
