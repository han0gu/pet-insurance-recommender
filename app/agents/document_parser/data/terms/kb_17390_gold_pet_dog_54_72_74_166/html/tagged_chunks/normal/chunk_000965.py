from langchain_core.documents import Document

chunk = Document(
    page_content=(". 대한민국 이외의 지역에서 발생한 사고 및 손해<br>부 가 설 명</p><br><table id='155' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>∙ 핵연료물질 : "
 '사용된</td><td>연료를 포함합니다.</td></tr><tr><td>∙ 핵연료물질에 의하여 아래의 치료비, 비용 또는 손해는 보상하지 '
 '않습니다.</td><td>오염된 물질 : 원자핵 분열 생성물을 포함합니다.</td></tr></tbody></table><br><h1 '
 "id='156'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000965',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
