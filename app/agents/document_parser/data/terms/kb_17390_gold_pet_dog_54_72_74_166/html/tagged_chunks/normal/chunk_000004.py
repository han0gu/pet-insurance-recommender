from langchain_core.documents import Document

chunk = Document(
    page_content=("되는 사람을 말합니다.</td></tr></tbody></table><p id='1' data-category='paragraph' "
 "style='font-size:14px'>54 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><p id='2' "
 "data-category='paragraph' style='font-size:14px'>2.</p><br><table id='3' "
 "style='font-size:14px'><thead><tr><td>지급사유 및 보상</td><td>관련"),
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
 'indexing': {'chunk_id': 'chunk_000004',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
