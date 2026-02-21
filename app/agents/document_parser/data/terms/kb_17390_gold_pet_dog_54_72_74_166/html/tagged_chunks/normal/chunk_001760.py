from langchain_core.documents import Document

chunk = Document(
    page_content=('간질성 폐질환 J84</td><td>관</td></tr><tr><td>폐 및 종격의 '
 "농양</td><td>J85</td></tr></tbody></table><br><h1 id='79' "
 "style='font-size:16px'>괴사성 질환</h1><br><p id='80' data-category='paragraph' "
 "style='font-size:16px'>농흉</p><br><p id='81' data-category='paragraph' "
 "style='font-size:16px'>J86</p><br><p id='82'"),
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
 'indexing': {'chunk_id': 'chunk_001760',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
