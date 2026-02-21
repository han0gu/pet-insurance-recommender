from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사는 이 계약의 체결, 유지, 보험금 지급 등<br>을 위하여 위 관계 법령에 따라 계약자 및 피보험자의 동의를 받아 다른 '
 '보험회사<br>및 보험관련단체 등에 개인정보를 제공할 수 있습니다.<br>\uf000 회사는 계약과 관련된 개인정보를 안전하게 관리하여야 '
 "합니다.</p><br><table id='176' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>관 련 법</td><td>규 "
 '개인정보보호법</td></tr><tr><td colspan="2">제17조(개인정보의 제공) :'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000324',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
