from langchain_core.documents import Document

chunk = Document(
    page_content=('기일까지</td><td>보험료가 납입되지</td><td></td><td>않을 경우, 회사가 계약자에게 '
 '보험료의</td></tr><tr><td colspan="4">납입을 재촉하는 것을 '
 "말합니다.</td></tr></tbody></table><br><p id='56' data-category='paragraph' "
 "style='font-size:16px'>제29조(보험료의 납입을 연체하여 해지된 계약의 부활(효력회복))</p><br><p "
 "id='57' data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000247',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
