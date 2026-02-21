from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>계약자는</h1><br><p id='104' data-category='paragraph' "
 "style='font-size:14px'>전환대상계약에 대하여 장애인전용보험으로의 전환을 취소할 수 있으며, 이</p><br><p "
 "id='105' data-category='paragraph' style='font-size:14px'>경우 전환취소 신청서를 회사에 "
 "제출하여야 합니다.</p><br><p id='106' data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_001436',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
