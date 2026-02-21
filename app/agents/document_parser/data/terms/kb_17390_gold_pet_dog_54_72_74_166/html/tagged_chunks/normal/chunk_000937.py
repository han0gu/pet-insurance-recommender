from langchain_core.documents import Document

chunk = Document(
    page_content=(". (자기부담금은 1일당 의료비에서 차감합니다.)</h1><br><table id='117' "
 'style=\'font-size:16px\'><thead><tr><td rowspan="2" colspan="2">구분</td><td '
 'colspan="2">1일당 보상한도액</td><td rowspan="2">질 연간 총 병 '
 '보상한도액</td></tr><tr><td>입/통원 중 수술을 하지 않은 날의 경우</td><td>입/통원 중 수술을 한 날의 '
 '경우</td></tr></thead><tbody><tr><td'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000937',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
