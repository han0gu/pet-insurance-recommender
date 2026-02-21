from langchain_core.documents import Document

chunk = Document(
    page_content=('specified</td><td>180Not specified</td><td>180Not '
 "specified</td></tr></tbody></table></figure><br><p id='116' "
 "data-category='paragraph' style='font-size:22px'>려동</p><p id='117' "
 "data-category='paragraph' style='font-size:14px'>\uf000</p><p id='118' "
 "data-category='paragraph' style='font-size:16px'>- 127"),
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
 'indexing': {'chunk_id': 'chunk_001281',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
