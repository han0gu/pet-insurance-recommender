from langchain_core.documents import Document

chunk = Document(
    page_content=('- 作)을 가하는 것을 말합니다.\n'
 '- \uf000 제1항의 수술에서 보건복지부 산하 신의료기술평가위원회(향후 제도 변경 시에는\n'
 '- 동 위원회와 동일한 기능을 수행하는 기관) 또는 이에 준하는 기관으로부터 안전\n'
 '- 성과 치료효과를 인정받은 최신 수술기법도 포함됩니다.\n'
 '- 82 -|  |\n'
 '| --- |\n'
 '| 용 어 풀 이 신의료기술평가위원회 |\n'
 '의료법 제54조(신의료기술평가위원회의 설치 등)에 의거 설치된 위원회로서 신| 의료기술에 관한 최고의 심의기구를 말합니다. |\n'
 '| --- |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000325',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
