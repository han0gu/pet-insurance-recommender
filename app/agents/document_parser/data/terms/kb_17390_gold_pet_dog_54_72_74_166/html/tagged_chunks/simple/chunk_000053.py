from langchain_core.documents import Document

chunk = Document(
    page_content=('회사가 추정하는</td><td></td><td></td><td>보험금의 50% 상당액을 가지급보험금으로 '
 '지급합니다.</td></tr><tr><td colspan="4">용 어 풀 이 가지급보험금 보험금 지급이 늦어지는 경우 회사가 지급할 '
 '것으로 예상되는 보험금의 일부를 먼저 지급하는 보험금 가지급제도에 따라 먼저 지급하는 보험금을 '
 "말합니다.</td></tr></tbody></table><br><p id='68' data-category='paragraph' "
 "style='font-size:16px'>\uf000 회사는 제1항의 규정에 정한"),
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
 'indexing': {'chunk_id': 'chunk_000053',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
