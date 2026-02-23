from langchain_core.documents import Document

chunk = Document(
    page_content=('2형 감염증, 코로나바이러스 감염증, 렙토스피라 감염증, 심상사상충 감염증, 광견병, 켄넬코프- 17. 기관협착, 누루관시술과 관련된 '
 '상해 또는 질병 치료에 대한 비용\n'
 '- 18. 아래의 유전적 또는 발달이상을 원인으로 하는 경우는 보상하지 않습니다.\n'
 '가. 뼈와 관절의 영역Wobbler증후군, 팔꿈치 관절형성부전, 팔꿈치 관절 척골 이탈, 팔꿈치 관절요골 이탈, 앞발'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000019',
              'chunk_char_len': 205,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
