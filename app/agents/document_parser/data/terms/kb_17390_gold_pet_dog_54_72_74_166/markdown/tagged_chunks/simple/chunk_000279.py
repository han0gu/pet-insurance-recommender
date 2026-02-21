from langchain_core.documents import Document

chunk = Document(
    page_content=('- 성과 치료효과를 인정받은 최신 수술기법도 포함됩니다.\n'
 '| 용 어 풀 | 이 신의료기술평가위원회 |\n'
 '| --- | --- |\n'
 '| 의료법 제54조(신의료기술평가위원회의 설치 등)에 의거 설치된 위원회로서 신 의료기술에 관한 최고의 심의기구를 말합니다. | 의료법 '
 '제54조(신의료기술평가위원회의 설치 등)에 의거 설치된 위원회로서 신 의료기술에 관한 최고의 심의기구를 말합니다. |\n'
 '# \uf000 제1항의 수술에서 아래에 정한 사항은 제외합니다.# 1. 흡인(吸引)- 2. 천자(穿刺) 등의 조치'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000279',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
