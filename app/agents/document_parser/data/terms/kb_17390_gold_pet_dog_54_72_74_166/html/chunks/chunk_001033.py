from langchain_core.documents import Document

chunk = Document(
    page_content=('어 풀</td><td>이</td></tr><tr><td colspan="2">∙ 아포퀠(Apoquel) 등의 JAK inhibitor, '
 '사이토포인트(Cytopoint) : 아토피 등 치료목적 성분 약물 ∙ 트릴로스탄(Trilostane) : 쿠싱증후군 등 치료목적 성분 '
 '약물 ∙ 피모벤단(Pimobendan) : 심장질환 치료목적 성분 약물 ∙ 크리스데살라진(Crisdesalazine)(제다큐어) : '
 '인지기능장애 치료목적 성분 약 물 \uf000 제1항 제6호에서 "특정재활치료Ⅱ"라 함은 수의사가 반려동물의 상해 또는'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
