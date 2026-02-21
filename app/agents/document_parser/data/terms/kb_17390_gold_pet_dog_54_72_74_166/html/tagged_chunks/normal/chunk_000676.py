from langchain_core.documents import Document

chunk = Document(
    page_content=('안전<br>성과 치료효과를 인정받은 최신 수술기법도 포함됩니다.<br>용 어 풀 이 신의료기술평가위원회</p><br><table '
 "id='233' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>의료법</td><td>제54조(신의료기술평가위원회의 "
 '설치 등)에 의거 설치된</td><td>위원회로서 신</td></tr><tr><td colspan="3">의료기술에 관한 최고의 '
 '심의기구를 말합니다'),
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
 'indexing': {'chunk_id': 'chunk_000676',
              'chunk_char_len': 245,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
