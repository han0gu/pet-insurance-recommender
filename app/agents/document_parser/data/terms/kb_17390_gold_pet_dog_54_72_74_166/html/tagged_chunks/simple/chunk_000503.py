from langchain_core.documents import Document

chunk = Document(
    page_content=('"연간"이란 계약일로부터 매1년 단위로 도래하는 계약해당일 전일까<br>지 기간을 의미합니다.</p><br><h1 id=\'232\' '
 "style='font-size:14px'>\uf000 제1항에서 치아파절진단비의 진단일자는 사고일을 기준으로 합니다.</h1><p "
 "id='233' data-category='list' style='font-size:14px'>제2조(보험금 지급에 관한 "
 '세부규정)<br>\uf000 제1조(보험금의 지급사유)의 치아파절진단비는 같은 상해를 직접적인 원인으<br>로 2가지 이상의 치아파절 '
 '발생시에는 1회에 한하여'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000503',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
