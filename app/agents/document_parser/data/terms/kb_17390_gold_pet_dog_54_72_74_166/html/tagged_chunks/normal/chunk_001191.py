from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>을 의미합니다.</p><p id='209' data-category='paragraph' "
 "style='font-size:16px'>제14조(합의․절충․중재․소송의 협조․대행 등)<br>\uf000 회사는 피보험자의 법률상 "
 '손해배상책임을 확정하기 위하여 피보험자가 피해자와<br>행하는 합의·절충·중재 또는 소송(확인의 소를 포함합니다)에 대하여 '
 "협조하거</p><br><p id='210' data-category='paragraph' style='font-size:16px'>나, "
 '피보험자를 위하여'),
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
 'indexing': {'chunk_id': 'chunk_001191',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
