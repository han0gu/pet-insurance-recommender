from langchain_core.documents import Document

chunk = Document(
    page_content=('. 성특<br>\uf000 회사는 납입최고(독촉)기간 안에 발생한 사고에 대하여 약정한 보험금을 지급합니 약</p><br><p '
 "id='32' data-category='paragraph' style='font-size:16px'>KB 금쪽같은 "
 "펫보험(강아지)(무배당)(26.01) 135</p><br><p id='33' data-category='paragraph' "
 "style='font-size:18px'>- 135 -</p><p id='34' data-category='paragraph' "
 "style='font-size:14px'>다"),
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
 'indexing': {'chunk_id': 'chunk_001389',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
