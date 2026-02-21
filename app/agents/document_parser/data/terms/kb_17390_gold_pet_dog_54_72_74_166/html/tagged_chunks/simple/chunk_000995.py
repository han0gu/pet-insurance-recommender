from langchain_core.documents import Document

chunk = Document(
    page_content=('colspan="2">특정재활치료Ⅱ</td><td>연간12회한</td><td>5만원</td></tr><tr><td '
 'colspan="2">항암약물치료</td><td>연간6회한</td><td>30만원</td></tr></tbody></table><br><p '
 "id='198' data-category='paragraph' style='font-size:14px'>\uf000 제1항에서 정한 "
 '반려동물의료비보험금이란 「1'),
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
 'indexing': {'chunk_id': 'chunk_000995',
              'chunk_char_len': 226,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
