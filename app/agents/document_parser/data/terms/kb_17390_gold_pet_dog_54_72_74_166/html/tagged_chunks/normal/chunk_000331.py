from langchain_core.documents import Document

chunk = Document(
    page_content=('보험금 등을 지급하지 못할 경우에는 예금자보호법에서 정 별<br>표<br>하는 바에 따라 그 지급을 보장합니다.</p><br><table '
 "id='182' style='font-size:16px'><thead></thead><tbody><tr><td>용 어 풀 "
 '이</td><td>예금자보호제도</td></tr><tr><td colspan="2">예금자보호제도란 예금보험공사에서 금융기관 등으로부터 '
 '미리 보험료를 받아 적 립해 두었다가 금융기관이 경영악화나 파산 등으로 예금을 지급할 수 없는 경우 해당 금융기관을 대신하여 '
 '해약환급금(또는'),
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
 'indexing': {'chunk_id': 'chunk_000331',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
