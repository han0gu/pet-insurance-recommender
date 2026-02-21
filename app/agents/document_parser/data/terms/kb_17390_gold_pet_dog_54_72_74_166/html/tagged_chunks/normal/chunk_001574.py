from langchain_core.documents import Document

chunk = Document(
    page_content=('bottom-right:(685,491)" /></figure><p id=\'80\' data-category=\'paragraph\' '
 "style='font-size:14px'>부 가 설 명</p><br><p id='81' data-category='paragraph' "
 'style=\'font-size:14px\'>골반뼈</p><figure id=\'82\'><img alt="" '
 'data-coord="top-left:(132,591); bottom-right:(725,923)" /></figure><br><p '
 "id='83'"),
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
 'indexing': {'chunk_id': 'chunk_001574',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
