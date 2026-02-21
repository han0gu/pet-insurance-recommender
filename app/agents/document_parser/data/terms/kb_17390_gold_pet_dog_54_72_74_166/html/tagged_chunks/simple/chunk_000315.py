from langchain_core.documents import Document

chunk = Document(
    page_content=('colspan="2">① 권리의 행사와 의무의 이행은 신의에 좇아 성실히 하여야 '
 "한다.</td></tr></tbody></table><h1 id='163' style='font-size:14px'>제47조(설명서 교부 "
 "및 보험안내자료 등의 효력)</h1><br><h1 id='164' style='font-size:14px'>\uf000 회사는 "
 "일반금융소비자에게 청약을 권유하거나</h1><br><p id='165' data-category='paragraph' "
 "style='font-size:14px'>일반금융소비자가 설명을"),
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
 'indexing': {'chunk_id': 'chunk_000315',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
