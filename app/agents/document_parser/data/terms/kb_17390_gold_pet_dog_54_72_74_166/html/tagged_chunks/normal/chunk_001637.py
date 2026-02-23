from langchain_core.documents import Document

chunk = Document(
    page_content=('흉복부장기 또는 비뇨생식기 기능에 뚜렷한 장해를 남긴 때</td><td>30</td></tr></tbody></table><br><p '
 "id='162' data-category='paragraph' style='font-size:14px'>15</p><br><h1 "
 "id='163' style='font-size:14px'>5) 흉복부장기 또는 비뇨생식기 기능에 약간의 장해를 남긴 때</h1><p "
 "id='164' data-category='paragraph' style='font-size:14px'>나.</p><br><p"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_001637',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
