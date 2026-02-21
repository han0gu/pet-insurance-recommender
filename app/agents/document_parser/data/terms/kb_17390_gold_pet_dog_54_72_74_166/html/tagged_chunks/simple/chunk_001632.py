from langchain_core.documents import Document

chunk = Document(
    page_content=('등을 따른다.</td></tr></thead><tbody><tr><td>부 가 설 명 발가락</td><td>보 통약 관 특별 약 관 '
 '<figure><img alt="" data-coord="top-left:(864,314); bottom-right:(1383,754)" '
 "/></figure> 별 표 법 ㆍ 규정</td></tr></tbody></table><p id='151' "
 "data-category='paragraph' style='font-size:14px'>KB 금쪽같은 "
 '펫보험(강아지)(무배당)(26.01)'),
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
 'indexing': {'chunk_id': 'chunk_001632',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
