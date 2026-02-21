from langchain_core.documents import Document

chunk = Document(
    page_content=('. 성<br>\uf000 피보험자가 보험기간 중 사망하고, 그 후에 제3조(천식지속상태의 정의 및 진단 특<br>확정)에서 정한 '
 '"천식지속상태"를 직접적인 원인으로 사망한 사실이 확인된 경우 약</p><br><p id=\'206\' '
 "data-category='paragraph' style='font-size:18px'>- 91 -</p><br><p id='207' "
 "data-category='paragraph' style='font-size:16px'>KB 금쪽같은 "
 "펫보험(강아지)(무배당)(26.01) 91</p><p id='208'"),
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
 'indexing': {'chunk_id': 'chunk_000658',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
