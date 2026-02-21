from langchain_core.documents import Document

chunk = Document(
    page_content=("id='13' style='font-size:16px'>\uf000 제1항에 따라 보험료 등의 감액 또는 증액시 환급금이 없거나 "
 "최초가입시 안내한</h1><br><table id='14' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>만기(해약)환급금보다</td><td>적거나 "
 '많아질 수 있습니다.</td></tr><tr><td colspan="2">용 어 풀 이 감액 보험료, 보험금, 계약자적립액 등을 산정하는 '
 '기준이 되는 보험가입금액을 계 약시 선택한 금액보다 적은 금액으로 줄이는 것'),
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
 'indexing': {'chunk_id': 'chunk_000201',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
