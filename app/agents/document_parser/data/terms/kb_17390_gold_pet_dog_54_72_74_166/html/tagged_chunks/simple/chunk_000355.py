from langchain_core.documents import Document

chunk = Document(
    page_content=("상태가 일정기간 이상 계속 될 때 이해관계가 있는 사</p><br><p id='8' data-category='paragraph' "
 "style='font-size:14px'>람의 청구에 의해 사망한 것으로 인정하고 신분이나 재산에 관한 모든 법적 관</p><br><p "
 "id='9' data-category='paragraph' style='font-size:14px'>계를 확정시키는 법원의 결정을 "
 "말합니다.</p><br><p id='10' data-category='paragraph' style='font-size:14px'>관 련"),
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
 'indexing': {'chunk_id': 'chunk_000355',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
